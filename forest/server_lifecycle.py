import forest.data as data


def on_server_loaded(server_context):
    data.on_server_loaded()


def on_session_destroyed(session_context):
    '''
    Function called when a session is closed 
    (e.g. tab closed or time out)
    '''
    if data.AUTO_SHUTDOWN:
        import sys 
        sys.exit('\033[1;31mThe session has ended - tab closed or timeout. \n\n --- Terminating the Forest progam and relinquishing control of port. ---\033[1;00m')