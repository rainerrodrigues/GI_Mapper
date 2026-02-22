# Substrate Blockchain Node

## Overview

Minimal Substrate blockchain node for storing cryptographic hashes of AI predictions and data for immutable audit trails.

## Features

- Custom pallet for data hash storage
- Hash verification
- Event emission for hash storage
- Simple consensus (Aura + GRANDPA)

## Architecture

```
substrate/
├── pallets/
│   └── data-hashes/     # Custom pallet for hash storage
├── runtime/             # Runtime configuration
└── node/                # Node implementation
```

## Quick Start

### Prerequisites

- Rust 1.70+
- Substrate dependencies

### Build

```bash
cd substrate
cargo build --release
```

### Run

```bash
./target/release/substrate-node --dev
```

## Custom Pallet: data-hashes

### Storage

- `DataHashes`: Map from hash (H256) to metadata (timestamp, data_type)

### Extrinsics

- `store_hash(hash, data_type)`: Store a new hash
- `verify_hash(hash)`: Verify if hash exists

### Events

- `HashStored(hash, data_type, timestamp)`
- `HashVerified(hash, exists)`

## Integration

The Rust backend connects to this blockchain via RPC to:
1. Submit hashes of predictions, clusters, scores
2. Verify data integrity
3. Provide audit trails

## Note

This is a simplified implementation for the platform. For production:
- Add proper consensus mechanism
- Implement permissioned access
- Add more sophisticated storage
- Implement proper key management
