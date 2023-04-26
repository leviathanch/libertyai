# Service files

First you've got to replace the place holder with the path to which you've
cloned the repo to.

    sed -e 's/LIBERTYAI_PATH/<path to repo>/g' *.service

Then you can copy the service files to the systemd folder, enable them and
start the services.


    sudo cp libertychatbot.service /etc/systemd/system
    sudo systemctl enable libertychatbot.service
    sudo systemctl start libertychatbot.service

