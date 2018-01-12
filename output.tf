output "url" {
  value = "http://${aws_instance.bokeh_server.public_ip}:8888"
}