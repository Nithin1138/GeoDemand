variable "aws_region" {
  type        = string
  description = "AWS deployment region (default: ap-south-1 Mumbai/Hyderabad for low latency in India)"
  default     = "ap-south-1"
}

variable "environment" {
  type        = string
  description = "Deployment environment name (production, staging, dev)"
  default     = "production"
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the dedicated GeoDemand VPC"
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  type        = list(string)
  description = "Availability zones for multi-AZ high availability"
  default     = ["ap-south-1a", "ap-south-1b"]
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for public subnets (ALB & NAT Gateway)"
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for private application subnets (ECS, RDS, Redis)"
  default     = ["10.0.11.0/24", "10.0.12.0/24"]
}

variable "domain_name" {
  type        = string
  description = "Primary domain name for the GeoDemand AI application"
  default     = "geodemand.ai"
}

variable "db_instance_class" {
  type        = string
  description = "RDS instance class for PostgreSQL 16 + PostGIS"
  default     = "db.t4g.xlarge"
}

variable "db_allocated_storage_gb" {
  type        = number
  description = "Allocated storage in GB for PostgreSQL RDS"
  default     = 100
}

variable "db_max_allocated_storage_gb" {
  type        = number
  description = "Maximum storage auto-scaling threshold in GB"
  default     = 500
}

variable "db_username" {
  type        = string
  description = "Master username for PostgreSQL database"
  default     = "geodemand_admin"
}

variable "db_password" {
  type        = string
  description = "Master password for PostgreSQL database (pass via TF_VAR_db_password or Secrets Manager)"
  sensitive   = true
  default     = "ChangeMeInProdSecure123!"
}

variable "redis_node_type" {
  type        = string
  description = "ElastiCache Redis node type"
  default     = "cache.t4g.medium"
}

variable "ecs_app_cpu" {
  type        = number
  description = "CPU units allocated for FastAPI task (1024 = 1 vCPU)"
  default     = 2048
}

variable "ecs_app_memory" {
  type        = number
  description = "Memory allocated for FastAPI task in MB"
  default     = 4096
}

variable "ecs_app_desired_count" {
  type        = number
  description = "Desired number of running ECS FastAPI task instances"
  default     = 2
}

variable "ecs_app_min_count" {
  type        = number
  description = "Minimum auto-scaling count for ECS tasks"
  default     = 2
}

variable "ecs_app_max_count" {
  type        = number
  description = "Maximum auto-scaling count for ECS tasks under peak load"
  default     = 10
}

