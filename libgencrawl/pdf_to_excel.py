
# Import Module 
import tabula
from datetime import datetime
import os

def timestr():
    now=datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H,%M,%S")
    return date_time

def pdftoexcel(path):
    dirname="excels_"+timestr()
    os.mkdir(dirname)
    os.chdir(dirname)
    files = os.listdir(Direc)
    files = [f for f in files if os.path.isfile(Direc+'/'+f)] #Filtering only the files.
    print(*files, sep="\n")
    #df = tabula.read_pdf("PDF File Path", pages = 1)[0]
    #df.to_excel('Excel File Path')


if __name__=='__main__':
    path=input(r"please input pdfs folder path and press enter: ")
    num=input('please input the number of pdfs you wanna transform and press enter: ')
    num=int(num)
    pdftoexcel(path,num)
