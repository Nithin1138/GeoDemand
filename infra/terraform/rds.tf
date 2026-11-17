resource "aws_db_subnet_group" "rds" {
  name        = "geodemand-${var.environment}-db-subnet-group"
  description = "Subnet group for GeoDemand PostGIS RDS cluster"
  subnet_ids  = aws_subnet.private[*].id

  tags = {
    Name = "geodemand-${var.environment}-db-subnet-group"
  }
}

resource "aws_db_parameter_group" "pg16" {
  name        = "geodemand-${var.environment}-pg16-params"
  family      = "postgres16"
  description = "Custom parameters for GeoDemand PostGIS 16"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }

  parameter {
    name  = "log_connections"
    value = "1"
  }

  parameter {
    name  = "log_disconnections"
    value = "1"
  }

  parameter {
    name  = "log_duration"
    value = "1"
  }

  tags = {
    Name = "geodemand-${var.environment}-pg16-params"
  }
}

resource "aws_db_instance" "postgres" {
  identifier                  = "geodemand-${var.environment}-postgres"
  engine                      = "postgres"
  engine_version              = "16.2"
  instance_class              = var.db_instance_class
  allocated_storage           = var.db_allocated_storage_gb
  max_allocated_storage       = var.db_max_allocated_storage_gb
  storage_type                = "gp3"
  storage_encrypted           = true
  db_name                     = "geodemand"
  username                    = var.db_username
  password                    = var.db_password
  port                        = 5432
  db_subnet_group_name        = aws_db_subnet_group.rds.name
  vpc_security_group_ids      = [aws_security_group.rds.id]
  parameter_group_name        = aws_db_parameter_group.pg16.name
  multi_az                    = var.environment == "production" ? true : false
  publicly_accessible         = false
  auto_minor_version_upgrade  = true
  backup_retention_period     = 14
  backup_window               = "18:00-19:00" # UTC (~11:30 PM IST)
  maintenance_window          = "Sun:20:00-Sun:21:00"
  deletion_protection         = var.environment == "production" ? true : false
  skip_final_snapshot         = var.environment == "production" ? false : true
  final_snapshot_identifier   = "geodemand-${var.environment}-final-snapshot"
  copy_tags_to_snapshot       = true

  tags = {
    Name = "geodemand-${var.environment}-postgres"
  }
}

