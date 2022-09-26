from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch, cm
from reportlab.graphics import renderPDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from svglib.svglib import svg2rlg
from datetime import datetime
from io import BytesIO
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import xarray as xr
from eofs.multivariate.standard import MultivariateEof
from shapely.geometry import mapping
from sklearn.cluster import KMeans
from scipy import signal
from obspy.signal.tf_misfit import cwt
import pymannkendall as mk
import dask
import time
import sys
import os
import threading
from time import sleep
from tqdm import tqdm


class Report(object):

    def __init__(self, report_name, first_line, line_width, margin):
        
        self.report = canvas.Canvas(f"{report_name}.pdf")
        self.last_line = first_line
        self.first_line = first_line
        self.line_width = line_width
        self.margin = margin


    def add_header(self):

        supp = self.line_width
        self.line_width = 15
        self.add_line("Report for radially constrained cluster",9)
        self.add_line("Author: Jacopo Grassi",9)
        self.add_line(f"Created {datetime.now()}",9)
        self.line_width = supp
        self.last_line = self.last_line - 30


    def add_line(self, text, font_size):

        if self.last_line-self.line_width < self.line_width:

            self.report.showPage()
            self.last_line = self.first_line

        self.report.setFont('Helvetica', font_size)
        self.report.drawString(self.margin,self.last_line,text)
        self.last_line = self.last_line-self.line_width


    def add_graph(self, fig, width, heigth):

        if self.last_line-heigth < self.line_width:

            self.report.showPage()
            self.last_line = self.first_line

        imgdata = BytesIO()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0)  # rewind the data
        imgdata=ImageReader(imgdata)

        self.report.drawImage(imgdata,self.margin,self.last_line-heigth,width=width,height=heigth)
        self.last_line = self.last_line-heigth

    def save(self):

        self.report.save()


    def early_exit(self, message, status):

        print(message)
        print(status)
        self.add_line('ERROR - ABORT',12)
        self.add_line(message,10)
        self.add_line(status,10)
        self.save()
        sys.exit(1)



class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1: 
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False




