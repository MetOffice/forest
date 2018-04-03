output "url" {
  value = "http://${aws_instance.forest_bokeh_server.public_ip}:80"
}