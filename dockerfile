FROM python:3.9
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY scripts/run.sh /
RUN chmod +x /run.sh
CMD ["/run.sh"]