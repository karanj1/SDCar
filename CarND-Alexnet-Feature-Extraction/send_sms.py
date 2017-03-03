def send_sms(msg):
    # we import the Twilio client from the dependency we just installed
    from twilio.rest import TwilioRestClient
    
    # the following line needs your Twilio Account SID and Auth Token
    client = TwilioRestClient("AC30533cc04155731bca1e5a3eff7c16b4", "1c6dfb9adc0e9190ac8cae2a13c6414c")
    
    # change the "from_" number to your Twilio number and the "to" number
    # to the phone number you signed up for Twilio with, or upgrade your
    # account to send SMS to any phone number
    client.messages.create(to="+18322712158", from_="+15629121012", 
                           body=msg)