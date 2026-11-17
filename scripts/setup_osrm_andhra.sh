#!/usr/bin/env bash
# ==============================================================================
# Setup OSRM Backend Container with Andhra Pradesh Extract
# Purpose: Prepares OSM road network graph for sub-5ms street-level routing
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OSRM_DATA_DIR="${REPO_ROOT}/data/osrm"

mkdir -p "${OSRM_DATA_DIR}"

OSRM_FILE="${OSRM_DATA_DIR}/andhra-pradesh-latest.osrm"
PBF_FILE="${OSRM_DATA_DIR}/andhra-pradesh-latest.osm.pbf"

echo "🗺️  GeoDemand AI — OSRM Andhra Pradesh Street Network Setup"
echo "📁 Target directory: ${OSRM_DATA_DIR}"

if [[ -f "${OSRM_FILE}" && -f "${OSRM_FILE}.hsgr" || -f "${OSRM_FILE}.mldgr" ]]; then
    echo "✅ OSRM graph is already built at ${OSRM_FILE}"
    echo "   You can now start the container using: docker compose up -d osrm"
    exit 0
fi

# 1. Download OpenStreetMap extract if not present
if [[ ! -f "${PBF_FILE}" ]]; then
    echo "📥 Downloading Andhra Pradesh / Southern India OSM extract from Geofabrik..."
    # India Southern zone extract covers Andhra Pradesh, Telangana, Karnataka, Tamil Nadu, Kerala
    GEOFABRIK_URL="https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf"
    
    if command -v curl &>/dev/null; then
        curl -fSL "${GEOFABRIK_URL}" -o "${PBF_FILE}" || {
            echo "⚠️ Could not download live Geofabrik extract directly. (Network or proxy restricted)."
            echo "   Please manually place 'andhra-pradesh-latest.osm.pbf' into ${OSRM_DATA_DIR}"
            exit 1
        }
    elif command -v wget &>/dev/null; then
        wget -O "${PBF_FILE}" "${GEOFABRIK_URL}"
    else
        echo "❌ Error: Neither curl nor wget is available."
        exit 1
    fi
fi

echo "⚙️  Extracting OSRM road graph with car profile..."
docker run --rm -t -v "${OSRM_DATA_DIR}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/andhra-pradesh-latest.osm.pbf

echo "⚙️  Partitioning OSRM multi-level Dijkstra graph..."
docker run --rm -t -v "${OSRM_DATA_DIR}:/data" osrm/osrm-backend osrm-partition /data/andhra-pradesh-latest.osrm

echo "⚙️  Customizing cells and speed weights..."
docker run --rm -t -v "${OSRM_DATA_DIR}:/data" osrm/osrm-backend osrm-customize /data/andhra-pradesh-latest.osrm

echo "✅ OSRM Graph successfully compiled!"
echo "   Run 'docker compose up -d osrm' to start the high-speed routing service on port 5000."
