output "url" {
  value = "http://${aws_instance.bokeh_server_dev20.public_ip}:8888"
}