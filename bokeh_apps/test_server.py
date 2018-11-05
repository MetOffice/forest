import unittest
import server


class TestParseArgs(unittest.TestCase):
    def test_show(self):
        args = server.parse_args([])
        result = args.show
        expect = False
        self.assertEqual(expect, result)

    def test_port(self):
        args = server.parse_args([])
        result = args.port
        expect = 5006
        self.assertEqual(expect, result)

    def test_port_given_flag(self):
        args = server.parse_args(["--port", "8888"])
        result = args.port
        expect = 8888
        self.assertEqual(expect, result)

    def test_allow_web_socket_origin(self):
        args = server.parse_args([
            "--allow-websocket-origin",
            "site.com"])
        result = args.allow_websocket_origin
        expect = ["site.com"]
        self.assertEqual(expect, result)

    def test_unused_session_lifetime(self):
        args = server.parse_args([])
        result = args.unused_session_lifetime
        expect = 15000
        self.assertEqual(expect, result)
