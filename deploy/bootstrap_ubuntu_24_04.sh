#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo bash deploy/bootstrap_ubuntu_24_04.sh" >&2
  exit 1
fi

if [[ "$(pwd)" != "/opt/grid-survival-research" ]]; then
  echo "Repository must be located at /opt/grid-survival-research for the bundled systemd units." >&2
  exit 1
fi

apt-get update
apt-get install -y ca-certificates curl gnupg git git-lfs chrony
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker chrony
git lfs install --system
git lfs pull

if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
fi

docker compose build
docker compose run --rm shadow-037-preflight

cp deploy/systemd/grid-survival-research.service /etc/systemd/system/
cp deploy/systemd/grid-survival-ops-hourly.service /etc/systemd/system/
cp deploy/systemd/grid-survival-ops-hourly.timer /etc/systemd/system/
cp deploy/systemd/grid-survival-daily.service /etc/systemd/system/
cp deploy/systemd/grid-survival-daily.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now grid-survival-research.service
systemctl enable --now grid-survival-ops-hourly.timer
systemctl enable --now grid-survival-daily.timer

docker compose ps
echo "Paper-only 037 infrastructure installed. No private API key or trading endpoint is used."
