terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6.0"
    }
  }

  # Production backend state configuration (S3 + DynamoDB state locking)
  # Uncomment and configure with your organization's backend bucket:
  # backend "s3" {
  #   bucket         = "geodemand-terraform-state-prod"
  #   key            = "state/geodemand-ai/terraform.tfstate"
  #   region         = "ap-south-1"
  #   dynamodb_table = "geodemand-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "GeoDemand-AI"
      ManagedBy   = "Terraform"
      Environment = var.environment
      Owner       = "Nithin-TechnicalLead"
      Version     = "2.1.0-prod"
    }
  }
}
