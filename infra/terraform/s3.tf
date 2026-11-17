resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# 1. Model Artifacts Bucket (Trained LightGBM champions, label encoders, schemas)
resource "aws_s3_bucket" "artifacts" {
  bucket        = "geodemand-${var.environment}-artifacts-${random_id.bucket_suffix.hex}"
  force_destroy = var.environment == "production" ? false : true

  tags = {
    Name        = "geodemand-${var.environment}-artifacts"
    ContentType = "ModelArtifacts"
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 2. Datasets Bucket (Parquet feature stores, multi-city partitions, Census GeoJSON)
resource "aws_s3_bucket" "datasets" {
  bucket        = "geodemand-${var.environment}-datasets-${random_id.bucket_suffix.hex}"
  force_destroy = var.environment == "production" ? false : true

  tags = {
    Name        = "geodemand-${var.environment}-datasets"
    ContentType = "FeatureStores"
  }
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 3. Amazon Elastic Container Registry (ECR) for GeoDemand Container Images
resource "aws_ecr_repository" "api" {
  name                 = "geodemand-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "geodemand-api-ecr"
  }
}

