import requests

url = 'http://localhost:5000/results'

r = requests.post(url, json={"regionc": 1,	"division": 4,	"reportable_domain": 2,	"hdd65": 6612,	"hdd30yr": 6365,	"cdd30yr": 813 ,
	"dollarel": 2055,	"dolelsph": 313,	"metromicro": 1,	"ur": 0,	"totrooms":11,	"heatroom":11,	"acrooms":11,	
	"totsqft": 6451})

print(r)
print(r.json())