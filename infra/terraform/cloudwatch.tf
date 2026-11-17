resource "aws_cloudwatch_log_group" "ecs_api" {
  name              = "/ecs/geodemand-${var.environment}-api"
  retention_in_days = 30

  tags = {
    Name = "geodemand-${var.environment}-api-logs"
  }
}

# Metric Alarm: High CPU Utilization (> 80% for 2 periods)
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  alarm_name          = "geodemand-${var.environment}-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = 60
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Alarm triggers when GeoDemand API CPU exceeds 80%"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.api.name
  }
}

# Metric Alarm: High Memory Utilization (> 85% for 2 periods)
resource "aws_cloudwatch_metric_alarm" "high_memory" {
  alarm_name          = "geodemand-${var.environment}-high-memory"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "MemoryUtilization"
  namespace           = "AWS/ECS"
  period              = 60
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "Alarm triggers when GeoDemand API memory usage exceeds 85%"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.api.name
  }
}

# Metric Alarm: 5XX Errors on ALB
resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "geodemand-${var.environment}-alb-5xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Triggers when GeoDemand ALB records more than 5 5XX errors in 1 minute"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }
}

