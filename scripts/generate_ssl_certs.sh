#!/usr/bin/env bash
# ==============================================================================
# Generate Self-Signed SSL Certificates for Dev / Staging NGINX Reverse Proxy
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SSL_DIR="${REPO_ROOT}/infra/nginx/ssl"

mkdir -p "${SSL_DIR}"

if [[ -f "${SSL_DIR}/fullchain.pem" && -f "${SSL_DIR}/privkey.pem" ]]; then
    echo "✅ SSL certificates already exist at ${SSL_DIR}"
    exit 0
fi

echo "🔐 Generating 2048-bit RSA self-signed SSL certificate for local / staging deployment..."
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "${SSL_DIR}/privkey.pem" \
    -out "${SSL_DIR}/fullchain.pem" \
    -subj "/C=IN/ST=Andhra Pradesh/L=Vijayawada/O=GeoDemand AI/OU=Engineering/CN=localhost" \
    2>/dev/null

chmod 600 "${SSL_DIR}/privkey.pem"
chmod 644 "${SSL_DIR}/fullchain.pem"

echo "✅ Self-signed SSL certificate successfully generated:"
echo "   - Certificate: ${SSL_DIR}/fullchain.pem"
echo "   - Private Key: ${SSL_DIR}/privkey.pem"

