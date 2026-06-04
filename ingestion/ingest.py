
import requests 
import os 
import json 
import csv 


class IngestDataset:
    def __init__(self,  dataset_config ,limit = 10000 , filename= 'top_papers.json'  ):
       self.name = filename
       self.limit = limit 
       self.datapath = dataset_config['dataset_dir']
       self.base_url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk?query="
       self.query =  "top papers data regarding lithium ion batteries"
       self.fields = "title,abstract,url,referenceCount,year"

    def BatteryData(self):
         
        #source :https://github.com/allenai/s2-folks/blob/main/examples/python/search_bulk/get_dataset.py

        print(f'Getting {self.limit} with query {self.query} from semantic scholar ')
        url = f"{self.base_url}{self.query}'&fields={self.fields}"
        print (url)

        # Define the file name
        file_path_name = os.path.join(self.datapath, self.name)
        if os.path.exists(file_path_name):
            print(f'Data already downloaded for {self.query}')
        else:
            url = f"{self.base_url}{self.query}'&fields={self.fields}"
            
            print(f'Using {url} to get scholarly articles')
            try: 
                r = requests.get(url).json()
                #print(f'Estimated to retrieve {r.get('total', 0)} documents')

            except e: 
                print(f'Issue with request {e} ')
                exit(1)

            with open(f"{file_path_name}", "a") as filehandler:
                json.dump(r, filehandler, indent = 4 )
            
            return 'Completed:' + self.query
            '''
            with open(f"{file_path_name}", mode = 'a', newline = '') as file :
                csv_writer = csv.writer(file)
            #write header to empty file to begin with 
            if os.stat(file_path_name).st_size == 0 :
                csv_writer.writerow(["paperId", "url","title", \
                                     "abstract","year", "referenceCount"])
            
            try: 
                while True:
                    if 'data' in r: #loop through each content of paper
                        for paper in r['data']:
                            print(paper)
                            csv_writer.writerow(paper.get("paperId"), \
                                            paper.get("url"), \
                                                paper.get("title"), \
                                                    paper.get("abstract"), \
                                                        paper.get("year"), \
                                                        paper.get("referenceCount")
                                                    )
                        retrieved +=len(r["data"])
                        print(f'Retrieved {retrieved} papers')
                    if 'token' not in r:
                        break 

                    try :
                        r = requests.get(f"{url}&token={r['token']}").json()
                        
                    except e:
                        print (f'Failed retrieving next page: {e}')
            

            except e1:
                print(f"something went wrong {e1}")    
            '''



        



    


    