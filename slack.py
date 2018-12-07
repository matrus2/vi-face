from slackclient import SlackClient
from pathlib import Path
'''
REQ:
pip install slackclient
'''


class Slack():

   def __init__(self):
       tokenPath = str(Path.home()) + "/token"
       tokenFile = open(tokenPath, "r")
       self.token = tokenFile.read()
       tokenFile.closed

   
   def send_slack_message_sandwich(self):
       sc = SlackClient(self.token)
       sc.api_call('chat.postMessage',
                   channel="GDYSFE6FP",
                   text="I am on 3rd floor :sandwich::bread::baguette_bread::croissant: ",
                   username="Mr Sandwich",
                   icon_emoji=':sandwich:'
                   )
    