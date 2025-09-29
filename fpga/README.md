# FPGA Backend (AWS F2/F1)

This directory contains the reference kernel used by the new `backend=fpga` execution mode. The implementation targets AWS **F2** instances (Xilinx UltraScale+ VU9P based) that expose the Shell runtime compatible with the legacy F1 developer kit. The design is intentionally conservative so it can be synthesized with the stock Vitis / SDAccel flow and acts as a starting point for more aggressive optimizations.

## Layout

- `kernels/double_sha256.cl` – OpenCL kernel that performs double SHA-256 across a contiguous nonce range. The host code expects the kernel symbol to be `double_sha256_fpga` and uses buffer arguments described below.
- `build.sh` – 简易 `v++` 封装脚本，读取环境变量构建 `.xclbin`。

## Building the Kernel

1. **Set up the AWS FPGA development kit** – install the AWS F1/F2 runtime (`aws-fpga` repo) and Xilinx Vitis 2023.1 or later. Source both the AWS shell environment and the Vitis tools before building.
2. **Synthesize**:
   ```bash
   export PLATFORM=/opt/aws/platform/aws-vu9p-f1-04261818/xdma/feature_aws_vnc_2/
   (cd repo/root && ./fpga/build.sh)
   ```
   The command above generates `double_sha256.awsxclbin`. Rename or copy it to the location you plan to reference from the service (for example `/opt/charms/double_sha256.awsxclbin`).
3. **Register the AFI (once per region)** – upload the `.awsxclbin` to S3 and register it with `aws ec2 create-fpga-image`. Alternatively, for development you can load the bitstream directly via `fpga-load-local-image`.

The supplied kernel is fully functional but prioritises clarity over ultimate throughput. It performs the nonce sweep sequentially inside the kernel. To scale performance, replicate the kernel (`--kernel_frequency`, `--link` pipeline) or refactor it into a streaming design with on-chip pipelining.

## Host Configuration

The Rust host exposes the following FPGA-specific knobs (all nested under `options.fpga` in the REST API):

| Field | Description | Default |
|-------|-------------|---------|
| `slot_id` | Physical slot id (0–7). Maps to the device index returned by `fpga-describe-local-image`. | `0` |
| `xclbin_path` | Absolute path to the compiled `.xclbin` file. **Required**. | — |
| `dma_buffer_bytes` | Reserved for future DMA-based streaming (currently unused). | `4 MiB` |
| `batches_per_exec` | Number of nonces the kernel processes per launch. Larger values reduce host/device round-trips. | `8192` |
| `streams` | Target number of independent compute units. The reference design runs a single work-item; set this to `1` unless you have replicated kernels. | `1` |

Example request payload:

```json
{
  "outpoint": "deadbeefcafebabe:0",
  "options": {
    "backend": "fpga",
    "binary_nonce": true,
    "total_nonce": 1000000000,
    "fpga": {
      "slot_id": 0,
      "xclbin_path": "/opt/charms/double_sha256.awsxclbin",
      "batches_per_exec": 1048576
    }
  }
}
```

### Runtime Prerequisites

- `fpga-mgmt` kernel module loaded (`sudo modprobe fpga_mgmt`).
- Xilinx runtime libraries available (`LD_LIBRARY_PATH` includes the Vitis/XRT `lib` directory).
- The `.xclbin` file must match the AWS shell version running on the machine.

### Performance Notes

- The baseline kernel uses a simple sequential loop. On an F2 instance you should see tens of MH/s. To reach GH/s you must unroll the nonce loop and pipeline the SHA-256 rounds (e.g. convert `sha256_digest` into a pair of streaming pipelines and feed multiple messages per clock).
- Increase `batches_per_exec` to amortise kernel launch overhead. The upper bound is limited by kernel runtime (watchdog timeout) and buffer sizes.
- Replicate the kernel (e.g. `v++ --connectivity.nk double_sha256_fpga:4`) and set `streams` accordingly after adding multiple command queues in the host if you need to drive several compute units concurrently.

### Debugging

- Inspect the AWS runtime logs with `sudo dmesg` and `/var/log/aws-fpga-runtime.log`.
- Enable OpenCL profiling by recompiling with `RUST_LOG=debug` and instrumenting the queue completion times.
- To validate correctness before flashing hardware, build in emulation (`v++ -t sw_emu`) and run the binary with `XCL_EMULATION_MODE=sw_emu`.
