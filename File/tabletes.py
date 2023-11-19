import cv2
import numpy as np
import time
import pytesseract as pyTes
from PIL import Image
import pandas as pd

pyTes.pytesseract.tesseract_cmd = 'D:/tesseract-ocr/tesseract.exe'
config = "--psm 4 --psm 6 --psm 12 --oem 1"
languages_ = "eng"



def optimizeDf(old_df: pd.DataFrame) -> pd.DataFrame:
    df = old_df[["left", "top", "width", "text"]]
    df['left+width'] = df['left'] + df['width']
    df = df.sort_values(by=['top'], ascending=True)
    df = df.groupby(['top', 'left+width'], sort=False)['text'].sum().unstack('left+width')
    df = df.reindex(sorted(df.columns), axis=1).dropna(how='all').dropna(axis='columns', how='all')
    df = df.fillna('')
    return df

def mergeDfColumns(old_df: pd.DataFrame, threshold: int = 10, rotations: int = 5) -> pd.DataFrame:
  df = old_df.copy()
  for j in range(0, rotations):
    new_columns = {}
    old_columns = df.columns
    i = 0
    while i < len(old_columns):
      if i < len(old_columns) - 1:
        # If the difference between consecutive column names is less than the threshold
        if any(old_columns[i+1] == old_columns[i] + x for x in range(1, threshold)):
          new_col = df[old_columns[i]].astype(str) + df[old_columns[i+1]].astype(str)
          new_columns[old_columns[i+1]] = new_col
          i = i + 1
        else: # If the difference between consecutive column names is greater than or equal to the threshold
          new_columns[old_columns[i]] = df[old_columns[i]]
      else: # If the current column is the last column
        new_columns[old_columns[i]] = df[old_columns[i]]
      i += 1
    df = pd.DataFrame.from_dict(new_columns).replace('', np.nan).dropna(axis='columns', how='all')
  return df.replace(np.nan, '')

def mergeDfRows(old_df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    new_df = old_df.iloc[:1]
    for i in range(1, len(old_df)):
        # If the difference between consecutive index values is less than the threshold
        if abs(old_df.index[i] - old_df.index[i - 1]) < threshold: 
            new_df.iloc[-1] = new_df.iloc[-1].astype(str) + old_df.iloc[i].astype(str)
        else: # If the difference is greater than the threshold, append the current row
            new_df = pd.concat([new_df,old_df.iloc[i]])
    return new_df.reset_index(drop=True)

def clean_df(df):
    # Remove columns with all cells holding the same value and its length is 0 or 1
    df = df.loc[:, (df != df.iloc[0]).any()]
    # Remove rows with empty cells or cells with only the '|' symbol
    df = df[(df != '|') & (df != '') & (pd.notnull(df))]
    # Remove columns with only empty cells
    df = df.dropna(axis=1, how='all')
    return df.fillna('')

def testocr():
    # pyTes.pytesseract.tesseract_cmd = 'D:/tesseract-ocr/tesseract.exe'
    img1 = Image.open("./Image/table1.png")
    img = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    # img = cv2.imread(img1)
    img = cv2.resize(img,None,fx=2,fy=2)
    
    # adth = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,85,11)
    himg,wimg,_ =  img.shape
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY )
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))
    # dialat = cv2.dilate(thresh,kernel,iterations=1)

    # textdata = pyTes.image_to_string(thresh,config=config , lang=languages_)

    # contours,hierachy = cv2.findContours(dialat,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
        # area = cv2.contourArea(cnt)
        # (x,y,w,h) = cv2.boundingRect(cnt)
        # if y< 50:
        # if area > 800:
            # (x,y,w,h) = cv2.boundingRect(cnt)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        # cv2.putText(img,textdata,(x,y+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    
    boxes = pyTes.image_to_boxes(thresh,config=config , lang=languages_ )
    data = pyTes.image_to_data(thresh,config=config , lang=languages_ , output_type='data.frame')
    data = data[['left','top','width','text']]
    data_imp_sort = optimizeDf(data)
    # print(data_imp_sort)
    # df_new_col = mergeDfColumns(data_imp_sort)
    # merged_row_df = mergeDfRows(df_new_col)
    # cleaned_df = clean_df(merged_row_df.copy())
    # print(cleaned_df)
    # textdata = pyTes.image_to_string(adth,config=config)
    # print(textdata)
    for b in boxes.splitlines():
        # print(b)
        b = b.split(' ')
        # print(b)
        t,x,y,w,h = b[0],int(b[1]),int(b[2]),int(b[3]),int(b[4])
        cv2.rectangle(img,(x,himg-y),(w,himg-h),(0,0,255),1)
        cv2.putText(img,t,(x,himg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    # print(pytesseract.image_to_string(gray))
    # cv2.imshow('thresh',thresh)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)


if __name__ == "__main__":
    # mainprogram()
    testocr()
    # vdo_ocr()
    # testmotor()
    # barcodecap()
    # Concept1()