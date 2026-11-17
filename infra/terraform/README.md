# 🌐 GeoDemand AI — Production Cloud-Native Terraform Infrastructure (Phase 15.4)

## Architecture Overview

This Terraform suite provisions the complete AWS production infrastructure for **GeoDemand AI (Movigo)** in the `ap-south-1` (India - Mumbai/Hyderabad) region.

```
                              Internet
                                 │
                   ┌─────────────┴─────────────┐
                   │ Application Load Balancer │ (:80 -> :443 redirect, TLS 1.3)
                   └─────────────┬─────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │    AWS ECS Fargate Cluster    │
                 │   (2-10 Tasks Auto-scaling)   │
                 └──────┬───────────────┬────────┘
                        │               │
        ┌───────────────┴──────┐ ┌──────┴──────────────┐
        │  PostgreSQL 16 + GIS │ │ ElastiCache Redis 7 │
        │  (RDS Multi-AZ 100GB)│ │ (Cluster Replication)│
        └──────────────────────┘ └─────────────────────┘
```

## Modular Resource Stack

| Module / File | Provisioned AWS Resources | Function |
|---|---|---|
| `main.tf` | Terraform Engine, AWS Provider (`~> 5.40`) | Cloud orchestration & backend locking |
| `vpc.tf` | Dedicated VPC `10.0.0.0/16`, 2 Public & 2 Private Subnets | Isolated network segmentation across 2 AZs |
| `security_groups.tf` | 5 Security Groups (ALB, ECS, RDS, Redis, OSRM) | Zero-trust defense with minimal port ingress |
| `rds.tf` | AWS RDS PostgreSQL 16.2 (`db.t4g.xlarge`) | PostGIS spatial table persistence & GIST index |
| `redis.tf` | AWS ElastiCache Redis 7 (`cache.t4g.medium`) | LRU response cache, idempotency & rate limits |
| `ecs.tf` | ECS Fargate Cluster, Task Definitions, Auto-Scaler | Container runtime for FastAPI, OSRM & Telegram |
| `alb.tf` | Application Load Balancer & Target Groups | Inbound HTTPS proxy, SSL offloading & healthchecks |
| `s3.tf` | 2 Encrypted S3 Buckets + ECR Container Registry | Model champion weights, feature stores & Docker images |
| `secrets.tf` | AWS Secrets Manager (`geodemand/production/api-secrets`) | Database credentials, API keys & webhook HMACs |
| `cloudwatch.tf` | Log Groups & CloudWatch Metric Alarms | CPU > 80%, RAM > 85%, ALB 5XX threshold monitoring |

## Quick Start Deployment

```bash
# 1. Initialize Terraform providers and backend
cd infra/terraform
terraform init

# 2. Review execution plan
cp terraform.tfvars.example terraform.tfvars
# (Edit terraform.tfvars with your production secrets)
terraform plan -out=tfplan

# 3. Apply infrastructure
terraform apply tfplan

# 4. View outputs
terraform output
```

## Security & Compliance
- **Zero Public Access:** Databases (RDS) and Caches (Redis) reside strictly in private subnets without public IPs.
- **KMS Encryption:** All S3 buckets and RDS storage volumes use AES-256 encryption at rest.
- **TLS In-Transit:** Redis ElastiCache enforces in-transit encryption (`rediss://`).
- **Disaster Recovery:** RDS automated backups with 14-day retention and multi-AZ automated failover.

