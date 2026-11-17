resource "aws_elasticache_subnet_group" "redis" {
  name        = "geodemand-${var.environment}-redis-subnet-group"
  description = "Subnet group for GeoDemand Redis 7 cache"
  subnet_ids  = aws_subnet.private[*].id

  tags = {
    Name = "geodemand-${var.environment}-redis-subnet-group"
  }
}

resource "aws_elasticache_parameter_group" "redis7" {
  name        = "geodemand-${var.environment}-redis7-params"
  family      = "redis7"
  description = "Custom parameters for GeoDemand Redis 7"

  parameter {
    name  = "maxmemory-policy"
    value = "volatile-lru"
  }

  parameter {
    name  = "notify-keyspace-events"
    value = "Ex"
  }

  tags = {
    Name = "geodemand-${var.environment}-redis7-params"
  }
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id          = "geodemand-${var.environment}-redis"
  description                   = "Redis 7 replication group for GeoDemand caching, idempotency & rate limits"
  node_type                     = var.redis_node_type
  num_cache_clusters            = var.environment == "production" ? 2 : 1
  parameter_group_name          = aws_elasticache_parameter_group.redis7.name
  port                          = 6379
  subnet_group_name             = aws_elasticache_subnet_group.redis.name
  security_group_ids            = [aws_security_group.redis.id]
  automatic_failover_enabled    = var.environment == "production" ? true : false
  multi_az_enabled              = var.environment == "production" ? true : false
  at_rest_encryption_enabled    = true
  transit_encryption_enabled    = true
  auto_minor_version_upgrade    = true
  maintenance_window            = "sun:22:00-sun:23:00"
  snapshot_retention_limit      = 7
  snapshot_window               = "19:00-20:00"

  tags = {
    Name = "geodemand-${var.environment}-redis"
  }
}
