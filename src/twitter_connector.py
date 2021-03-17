import tweepy


class TwitterConnector:
    SECRETS_PATH = "../secrets/twitter_keys.txt"

    def __init__(self):
        keys = self.__read_secrets()
        self.api = self.__setup_access(keys)

    def __read_secrets(self):
        d = {}
        with open(self.SECRETS_PATH) as f:
            for line in f:
                (key, val) = line.split()
                d[key] = val
        return d

    def __setup_access(self, keys):
        auth = tweepy.OAuthHandler(keys["api_key"], keys["api_key_secret"])
        auth.set_access_token(keys["access_token"], keys["access_token_secret"])

        return tweepy.API(auth)

    def get_api(self):
        return self.api
