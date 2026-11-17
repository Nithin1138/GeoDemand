output "vpc_id" {
  description = "ID of the GeoDemand dedicated VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "alb_dns_name" {
  description = "Public DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "rds_endpoint" {
  description = "PostgreSQL 16 + PostGIS RDS cluster endpoint"
  value       = aws_db_instance.postgres.endpoint
}

output "redis_primary_endpoint" {
  description = "Primary endpoint address of the Redis 7 ElastiCache cluster"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "ecr_repository_url" {
  description = "ECR Repository URL for container image pushes"
  value       = aws_ecr_repository.api.repository_url
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for trained model artifacts and champions"
  value       = aws_s3_bucket.artifacts.id
}

output "s3_datasets_bucket" {
  description = "S3 bucket for versioned Parquet feature stores"
  value       = aws_s3_bucket.datasets.id
}

output "secrets_manager_secret_arn" {
  description = "ARN of Secrets Manager holding API credentials"
  value       = aws_secretsmanager_secret.app_secrets.arn
}
