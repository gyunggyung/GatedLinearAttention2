#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "This legacy script now delegates to the local pure GDN-2 FineWeb-Edu 100BT launcher."
echo "Use scripts/pretrain_fineweb_edu_100bt_gdn2.sh directly for new runs."

exec "${SCRIPT_DIR}/pretrain_fineweb_edu_100bt_gdn2.sh" "$@"
