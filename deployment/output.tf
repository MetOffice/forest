output "url" {
  value = "http://${aws_instance.forest_bokeh_server_dev21.public_ip}:8888"
}