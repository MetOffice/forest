"""Utilities to help with testing"""
import os


def remove_after(test_case, file_name):
    def cleanup():
        if os.path.exists(file_name):
            os.remove(file_name)
    test_case.addCleanup(cleanup)
    return file_name
