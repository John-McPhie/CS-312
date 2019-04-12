import os
def notify(text="", title="Notification", sound=""):
    if sound.lower() == "none":
        cmd = 'osascript -e \'display notification "{}" with title "{}"\''.format(text,title)
    else:
        cmd = 'osascript -e \'display notification "{}" with title "{}" sound name "{}"\''.format(text,title,sound)
    os.system(cmd)

def app_notify(text="", title="Notification", app="System Events"):
    cmd = 'osascript -e \'tell app "{}" to display dialog "{}" with title "{}"\''.format(app, text, title)
    result = os.system(cmd)
    return True if result == 0 else False

def list_sounds():
    cmd = 'ls /System/Library/Sounds | sed -e \'s/\..*$//\''
    os.system(cmd)
