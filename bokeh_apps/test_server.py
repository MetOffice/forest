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

    def test_unused_session_lifetime_default(self):
        self.check([], "unused_session_lifetime", 15000)

    def test_unused_session_lifetime_flag(self):
        self.check(["--unused-session-lifetime", "1000"], "unused_session_lifetime", 1000)

    def test_keep_alive_default(self):
        self.check([], "keep_alive", 37000)

    def test_keep_alive_flag(self):
        self.check(["--keep-alive", "1000"], "keep_alive", 1000)

    def check(self, argv, attr, expect):
        args = server.parse_args(argv)
        result = getattr(args, attr)
        self.assertEqual(expect, result)
