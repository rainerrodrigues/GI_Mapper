# Substrate Node Setup Guide

## Overview

This guide explains how to set up and run the Substrate blockchain node for the AI GIS Platform.

## Prerequisites

### System Requirements
- Linux or macOS (Windows via WSL2)
- 4GB RAM minimum
- 10GB disk space

### Software Requirements
- Rust 1.70+ with wasm32 target
- Substrate dependencies

## Installation

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Add WebAssembly Target

```bash
rustup target add wasm32-unknown-unknown
```

### 3. Install Substrate Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y build-essential git clang curl libssl-dev llvm libudev-dev make protobuf-compiler
```

#### macOS
```bash
brew install openssl cmake llvm
```

## Quick Start (Development Mode)

For development and testing, you can use a pre-built Substrate node template:

### Option 1: Use Substrate Node Template

```bash
# Clone substrate node template
git clone https://github.com/substrate-developer-hub/substrate-node-template
cd substrate-node-template

# Build the node
cargo build --release

# Run in development mode
./target/release/node-template --dev
```

The node will be available at:
- RPC: http://localhost:9933
- WebSocket: ws://localhost:9944

### Option 2: Build Custom Node (Future)

```bash
cd substrate
cargo build --release
./target/release/substrate-node --dev
```

## Configuration

### Backend Connection

Update backend `.env` file:

```env
BLOCKCHAIN_NODE_URL=http://localhost:9933
BLOCKCHAIN_WS_URL=ws://localhost:9944
```

### Node Configuration

Create `config.json`:

```json
{
  "rpc_port": 9933,
  "ws_port": 9944,
  "p2p_port": 30333,
  "chain": "dev",
  "base_path": "/tmp/substrate-node"
}
```

## Running the Node

### Development Mode

```bash
./target/release/node-template --dev --tmp
```

Flags:
- `--dev`: Development mode with pre-funded accounts
- `--tmp`: Use temporary directory (data deleted on exit)

### Production Mode

```bash
./target/release/node-template \
  --chain=local \
  --base-path=/var/lib/substrate \
  --rpc-external \
  --ws-external \
  --rpc-cors=all
```

Flags:
- `--chain=local`: Use local chain spec
- `--base-path`: Data directory
- `--rpc-external`: Allow external RPC connections
- `--ws-external`: Allow external WebSocket connections
- `--rpc-cors=all`: Allow CORS (configure properly for production)

## Verifying Installation

### Check Node Status

```bash
curl -H "Content-Type: application/json" \
  -d '{"id":1, "jsonrpc":"2.0", "method": "system_health"}' \
  http://localhost:9933
```

Expected response:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "peers": 0,
    "isSyncing": false,
    "shouldHavePeers": false
  },
  "id": 1
}
```

### Test from Backend

```bash
# Start backend
cd backend
cargo run

# Test blockchain status
curl http://localhost:3000/api/v1/blockchain/status
```

## Custom Pallet Integration (Future)

To integrate the custom `data-hashes` pallet:

### 1. Add Pallet to Runtime

Edit `runtime/src/lib.rs`:

```rust
// Add pallet to construct_runtime! macro
construct_runtime!(
    pub enum Runtime where
        Block = Block,
        NodeBlock = opaque::Block,
        UncheckedExtrinsic = UncheckedExtrinsic
    {
        // ... other pallets
        DataHashes: pallet_data_hashes,
    }
);
```

### 2. Configure Pallet

```rust
impl pallet_data_hashes::Config for Runtime {
    type RuntimeEvent = RuntimeEvent;
}
```

### 3. Rebuild Runtime

```bash
cargo build --release
```

## Monitoring

### View Logs

```bash
# Follow logs
tail -f /var/log/substrate-node.log

# View with journalctl (if using systemd)
journalctl -u substrate-node -f
```

### Check Block Height

```bash
curl -H "Content-Type: application/json" \
  -d '{"id":1, "jsonrpc":"2.0", "method": "chain_getHeader"}' \
  http://localhost:9933
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :9933

# Kill process
kill -9 <PID>
```

### Build Errors

```bash
# Clean build
cargo clean

# Update dependencies
cargo update

# Rebuild
cargo build --release
```

### Connection Refused

Check if node is running:
```bash
ps aux | grep substrate
```

Check firewall:
```bash
# Allow RPC port
sudo ufw allow 9933

# Allow WebSocket port
sudo ufw allow 9944
```

## Production Deployment

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM paritytech/ci-linux:production as builder

WORKDIR /substrate
COPY . .
RUN cargo build --release

FROM debian:buster-slim
COPY --from=builder /substrate/target/release/node-template /usr/local/bin

EXPOSE 9933 9944 30333
CMD ["node-template", "--chain=local", "--rpc-external", "--ws-external"]
```

Build and run:

```bash
docker build -t substrate-node .
docker run -p 9933:9933 -p 9944:9944 substrate-node
```

### Using Systemd

Create `/etc/systemd/system/substrate-node.service`:

```ini
[Unit]
Description=Substrate Node
After=network.target

[Service]
Type=simple
User=substrate
ExecStart=/usr/local/bin/node-template --chain=local --base-path=/var/lib/substrate
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable substrate-node
sudo systemctl start substrate-node
sudo systemctl status substrate-node
```

## Security

### Production Checklist

- [ ] Use proper chain specification (not --dev)
- [ ] Configure firewall rules
- [ ] Use TLS for RPC/WebSocket
- [ ] Implement authentication
- [ ] Regular backups of chain data
- [ ] Monitor node health
- [ ] Keep software updated
- [ ] Use dedicated user account
- [ ] Restrict CORS properly
- [ ] Enable rate limiting

## Resources

- [Substrate Documentation](https://docs.substrate.io/)
- [Substrate Node Template](https://github.com/substrate-developer-hub/substrate-node-template)
- [Polkadot Wiki](https://wiki.polkadot.network/)
- [Substrate Stack Exchange](https://substrate.stackexchange.com/)

## Support

For issues or questions:
1. Check logs: `/var/log/substrate-node.log`
2. Review Substrate documentation
3. Check GitHub issues
4. Ask on Substrate Stack Exchange

## Next Steps

1. Start node in development mode
2. Test connection from backend
3. Submit test hash
4. Verify hash on blockchain
5. Monitor node performance
6. Plan production deployment
