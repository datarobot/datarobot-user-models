FROM datarobot/mirror_chainguard_datarobot.com_python-fips:3.11-dev

USER root
RUN cp /etc/apk/world /etc/apk/world.orig
RUN sed 's/=.*//' /etc/apk/world.orig > /etc/apk/world
RUN apk update; apk upgrade -v
