# ALB Security Group (Public Ingress)
resource "aws_security_group" "alb" {
  name        = "geodemand-${var.environment}-alb-sg"
  description = "Controls public inbound traffic to GeoDemand Application Load Balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Allow HTTPS from anywhere"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Allow HTTP from anywhere (redirected to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound traffic to private targets"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "geodemand-${var.environment}-alb-sg"
  }
}

# ECS Application Security Group (Private)
resource "aws_security_group" "ecs_app" {
  name        = "geodemand-${var.environment}-ecs-app-sg"
  description = "Allows inbound traffic only from the ALB to FastAPI container"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Allow port 8000 from ALB only"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "Allow all outbound traffic for API calls, package installs & monitoring"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "geodemand-${var.environment}-ecs-app-sg"
  }
}

# RDS PostgreSQL 16 + PostGIS Security Group
resource "aws_security_group" "rds" {
  name        = "geodemand-${var.environment}-rds-sg"
  description = "Allows PostgreSQL traffic strictly from ECS application tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from ECS App tasks"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_app.id]
  }

  egress {
    description = "No direct outbound needed"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "geodemand-${var.environment}-rds-sg"
  }
}

# Redis ElastiCache Security Group
resource "aws_security_group" "redis" {
  name        = "geodemand-${var.environment}-redis-sg"
  description = "Allows Redis cache access strictly from ECS application tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Redis from ECS App tasks"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_app.id]
  }

  egress {
    description = "No outbound needed"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "geodemand-${var.environment}-redis-sg"
  }
}

# OSRM Routing Engine Security Group
resource "aws_security_group" "osrm" {
  name        = "geodemand-${var.environment}-osrm-sg"
  description = "Allows OSRM street-routing traffic from ECS application tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "OSRM port 5000 from ECS App"
    from_port       = 5000
    to_port         = 5000
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_app.id]
  }

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "geodemand-${var.environment}-osrm-sg"
  }
}

