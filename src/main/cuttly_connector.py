import urllib
import requests


class CuttlyConnector:
    __SECRETS_PATH_CUTTLY = "../../secrets/cuttly_keys.txt"

    def __init__(self):
        self.__key = self.__read_secrets()

    def __read_secrets(self):
        with open(self.__SECRETS_PATH_CUTTLY) as f:
            for line in f:
                key = line
        return key

    def shorten_link(self, link):
        url = urllib.parse.quote(link)
        name = ''
        r = requests.get('http://cutt.ly/api/api.php?key={}&short={}&name={}'.format(self.__key, url, name))
        response = eval(r.text)

        if response["url"]["status"] == 7:
            short_link = response["url"]["shortLink"]
            short_link = str(short_link).replace("\\", "")
            return short_link
        else:
            return None


def main():
    c = CuttlyConnector()

    link = c.shorten_link("dafdas")

    print(link)


if __name__ == '__main__':
    main()
