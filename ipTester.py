import subprocess
import time

def check_device_status(ip_address):
    # Ping the device's IP address
    result = subprocess.run(['ping', '-c', '1', ip_address], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    
    # Check the return code to determine if the device is online or offline
    if result.returncode == 0:
        status = "Online"
        print("Online")
    else:
        status = "Offline"
        print("Offline")
    
    return status

def main():
    ip_address = input("Enter the IP address of the device: ")

    # Loop indefinitely to continually check the device status
    while True:
        status = check_device_status(ip_address)
        
        # append the status to a text file
        with open('device_status_skysys.txt', 'a') as file:
            file.write(f"Device Status: {status}\n")

        # Wait for a specified interval before checking again
        time.sleep(2)  # Adjust the interval as needed

if __name__ == "__main__":
    main()
