resource "aws_secretsmanager_secret" "app_secrets" {
  name        = "geodemand/${var.environment}/api-secrets"
  description = "Production runtime API keys, payment webhook secrets & Telegram bot token"

  recovery_window_in_days = 0 # Immediate purge on delete in non-prod

  tags = {
    Name = "geodemand-${var.environment}-secrets"
  }
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id

  secret_string = jsonencode({
    DATABASE_URL            = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.postgres.endpoint}/geodemand"
    REDIS_URL               = "rediss://${aws_elasticache_replication_group.redis.primary_endpoint_address}:6379"
    SECRET_KEY              = "prod-secret-key-geodemand-production-change-me"
    API_KEY                 = "geodemand-prod-live-key"
    TELEGRAM_BOT_TOKEN      = ""
    RAZORPAY_WEBHOOK_SECRET = "rzp_sec_prod_live"
    PAYTM_WEBHOOK_SECRET    = "paytm_sec_prod_live"
    PHONEPE_WEBHOOK_SECRET  = "phonepe_sec_prod_live"
  })
}

