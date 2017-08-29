import xlrd
import numpy as np
import xlwt

workbook = xlrd.open_workbook('NYgas')
sh = workbook.sheet_by_index(1)




"""
need to calculate avg gas price for each year 1987-2016
cells B11-B370
1987: B11-B22
"""



prices = []

for i in range(10,369):
    
    value = sh.cell_value(i,1)
    prices.append(value)

yearly_averages = []

start = 0
for i in range(0,30):
    
    avg = np.mean(prices[start:start+12])
    yearly_averages = np.append(yearly_averages,avg)
    start += 12
    

finaldata = np.zeros((30,2))
finaldata[0:30,0]= 1987+ np.arange(0,30)
finaldata[0:30,1] = yearly_averages






    

workbook2 = xlsxwriter.Workbook('YearlyAverages.xlsx')
worksheet = workbook2.add_worksheet()

row=1
col=0

worksheet.write(col,0,'Year')
worksheet.write(col,1,'Average Gasoline Price')

for item, price in (finaldata):
    worksheet.write(row,col,item)
    worksheet.write(row,col+1,price)
    row += 1
    
workbook2.close()





    
    
    



    










