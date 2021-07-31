from tkinter import *
from tkinter import ttk
import pyodbc
import pandas as pd

class CutTextbox(Entry):
        def __init__(self,master,**kw):
            Entry.__init__(self,master = master,**kw)
            self.bind('<Return>',self.cutting)
            # self.bind('<Enter>',self.hoveron)

        def cutting(self,e):
            self.keepdata = e.widget.get()
            if(len(self.keepdata) > 10):
                self.delete(0,END)
                self.insert(END,self.keepdata[0:3])

def checkdata(e):
    # e.widget.get
    # row = en.grid_info()['row']
    # print(row)
    rec = e.widget.get()
    # print(rec)
    index = int(e.widget.name)
    # print(index)
    tar = partitem[index]["text"]
    print(index,rec,tar)
    if rec == tar:
        stat[index]["text"] = "OK"
        barq[index]["bg"] = "lightgreen"
        partitem[index]["bg"] = "lightgreen"
        stat[index]["bg"] = "lightgreen"
    else:
        stat[index]["text"] = "NG"
        barq[index]["bg"] = "indianred"
        partitem[index]["bg"] = "indianred"
        stat[index]["bg"] = "indianred"

con_string = 'Driver={SQL Server};Server=127.0.0.1;Database = test;UID = client1;PWD = nutert0405;'
def getdatabase(e):
    model = getmodel.get()
    sqlstr = f"""
                select * 
                from [test].[dbo].[pcbtest] 
                where model = '{model}'
            """
    conn = pyodbc.connect(con_string)
    df = pd.read_sql(sqlstr,conn)
    for i in range(11):
        stat[i]["text"] = ""
        znum[i]["text"] = ""
        partitem[i]["text"] = ""

        znum[i]["bg"] ="white"
        partitem[i]["bg"] = "white"
        barq[i]["bg"] = "white"

        barq[i]["state"] = "normal"
    j = 0
    newdf = df.values.tolist()
    for i in range(11):
        try:
            if i == int(newdf[j][0])-1:
                znum[i]["text"] = newdf[j][0]
                partitem[i]["text"] = newdf[j][1]
                j+=1
            else:
                znum[i]["text"] = i +1
                partitem[i]["text"] = ""

                znum[i]["bg"] ="dimgrey"
                partitem[i]["bg"] = "dimgrey"
                barq[i]["bg"] = "dimgrey"

                barq[i]["state"] = "readonly"
        except Exception as er :
            print('Error -> {}'.format(er))
            znum[i]["bg"] ="dimgrey"
            barq[i]["bg"] = "dimgrey"
            partitem[i]["bg"] = "dimgrey"
            barq[i]["state"] = "readonly"


def clearall(e):
    for i in range(11):
        stat[i]["text"] = ""
        znum[i]["text"] = ""
        barq[i]["text"] = ""
        partitem[i]["text"] = ""
        
root = Tk()
root.geometry("600x600")
style = ttk.Style()
style.configure("TNotebook.Tab",font = "consolas 15") 

tab_parent = ttk.Notebook(root)
tab1 = ttk.Frame(root)
tab2 = ttk.Frame(root)

left_frame = ttk.Frame(tab1)
left_frame.pack(expand = 1,fill = BOTH)

right_frame = ttk.Frame(tab1)
right_frame.pack(expand = 1,fill = BOTH)

tab_parent.add(tab1,text = "MODEL CHANGE")
tab_parent.add(tab2,text = "SERIES CHANGE")
tab_parent.pack(expand = 1,fill = BOTH)

znum = []
barq = []
partitem = []
stat = []

bt1 = Button(left_frame,text = "CLEAR ALL",font = "consolas 20")
bt1.bind('<Button-1>',clearall)
bt1.grid(row = 0 , column = 1)

getmodel = StringVar()
tb1 = Entry(left_frame,textvariable = getmodel,relief='solid',width = 10,font = "consolas 20")
tb1.bind('<Return>',getdatabase)
tb1.grid(row = 0 , column = 2)


for i in range(11):
    
    zid = Label(left_frame,relief='solid',width = 3,borderwidth=1,font = "consolas 20")
    zid.name = f"{i}"
    zid.grid(row = i+1,column = 0,sticky = "NEWS")
    
    partname = Label(left_frame,relief='solid' ,width = 10,borderwidth=1,font = "consolas 20")
    partname.name = f"{i}"
    partname.grid(row = i+1,column = 1,sticky = "NEWS")

    barcode = Entry(left_frame,relief='solid',width = 10 ,borderwidth=1,font = "consolas 20",justify = "center")
    barcode.name =f"{i}"
    barcode.bind('<Return>',checkdata)
    barcode.grid(row = i+1,column = 2,sticky = "NEWS")

    status = Label(left_frame,relief='solid',width = 5,borderwidth=1 ,font = "consolas 20")
    status.name = f"{i}"
    status.grid(row = i+1,column = 3,sticky = "NEWS")

    znum.append(zid)
    partitem.append(partname)
    barq.append(barcode)
    stat.append(status)

root.mainloop()