import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from tqdm.auto import tqdm
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import fuckit
import time
import argparse

from pyvirtualdisplay import Display
import time

parser = argparse.ArgumentParser()
options = Options()
display = Display(size=(100, 100))  
display.start()

#config web
options.add_argument('--headless')
options.add_experimental_option('excludeSwitches', ['enable-logging'])
# create instance of Chrome webdriver
automation_dir = "/home/xps/educate/code/hust/XQA/crawler/automatic_post"
driver_path = f"{automation_dir}/chromedriver"
driver=webdriver.Chrome(executable_path=driver_path, chrome_options=options,  desired_capabilities={"page_load_strategy": "none"})


#args
parser.add_argument('--from_city', default="Hai Phong")
parser.add_argument('--to_city', default="Vinh")
parser.add_argument('--num_class', default=0)
parser.add_argument('--adult', default=1)
parser.add_argument('--child', default=0)
parser.add_argument('--infant', default=0)
parser.add_argument('--num_images', default=5, type=int)


def send_request(from_city='Hai Phong', to_city='Vinh', num_class=0, pass_dict={'adult': 2, 'child': 2, 'infant': 0}):
    url = "https://www.traveloka.com/en-vn/"
    driver.get(url)
    print("GET URL!!!!")
    driver.save_screenshot(f"{automation_dir}/traveloka.png")
    #elements
    origin = driver.find_element(By.CSS_SELECTOR, "input[placeholder=Origin]")
    origin.clear()
    driver.implicitly_wait(1000)
    origin.send_keys(from_city)
    # xbox = WebDriverWait(driver, 3000).until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[1]/div[5]/div[2]/div[1]/div[2]/div/div[3]/div/div[2]/div/div[1]/div[1]/div[1]/div[2]/div/div/div[2]/div")))
    xbox = driver.find_element(By.CSS_SELECTOR, "div[data-cell-content='Hai Phong, Vietnam'")
    xbox.click()
    

    dest = driver.find_element(By.CSS_SELECTOR, "input[placeholder=Destination]")
    dest.clear()
    dest.send_keys(to_city)
    xbox = driver.find_element(By.CSS_SELECTOR, "div[data-cell-content='Vinh, Vietnam'")
    xbox.click()
    passenger = driver.find_element(By.CSS_SELECTOR, "input[aria-labelledby=flight_passengers]")
    passenger.click()
    for key, value in pass_dict.items():
        button = driver.find_element(By.CSS_SELECTOR, f'div[role=button][data-testid="passengers-stepper-plus-{key}"]')
        for i in range(int(value)):
            button.click()
    done_button = driver.find_element(By.CSS_SELECTOR, 'div[class="css-18t94o4 css-1dbjc4n r-kdyh1x r-1loqt21 r-10paoce r-5njf8e r-1otgn73 r-lrvibr"]')
    done_button.click()
    #seat class
    seat_class = driver.find_element(By.CSS_SELECTOR, "div[aria-haspopup=listbox]")
    seat_class.click()
    driver.find_elements(By.CSS_SELECTOR, "div[role=option]")[int(num_class)].click()

    #search 
    driver.save_screenshot(f"{automation_dir}/request.png")
    time.sleep(1)
    driver.find_elements(By.CSS_SELECTOR, "div[class='css-901oao css-bfa6kz r-jwli3a r-1sixt3s r-1o4mh9l r-b88u0q r-f0eezp r-q4m81j']")[1].click()
    
    #wait and take screenshot
    print("FILLED THE FORM !!!")
    num_images = args.num_images
    for i in range(num_images):
        time.sleep(3)
        window_height = driver.get_window_size()['height']
        print(window_height)
        driver.execute_script(f"window.scrollTo(0, {window_height*(i+1)});")
        time.sleep(1)
        driver.save_screenshot(f"{automation_dir}/results/temp{i}.png")


    
    
if __name__ == "__main__":

    args = parser.parse_args()
    send_request(args.from_city, args.to_city, args.num_class, pass_dict={'adult': args.adult, 'child': args.child, 'infant': args.infant})