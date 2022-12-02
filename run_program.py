import sys

def run_app():
    dictionary = {}
    if len(sys.argv) == 1:
        dictionary = dict(host='0.0.0.0', port='5000')
    elif len(sys.argv) == 2:
        if "=" in sys.argv[1]:
            split = sys.argv[1].split('=')
            dictionary.update({split[0]:split[1]})
        else:
            dictionary.update({'host':sys.argv[1]})
    else:
         for i, x in enumerate(sys.argv[1:3]):
            if i == 0:
                if "=" in x:
                    split = x.split('=')
                    dictionary.update({split[0]:split[1]})
                else:
                    dictionary.update({'host':sys.argv[1]})
            else:
                if "=" in x:
                    split = x.split('=')
                    dictionary.update({split[0]:split[1]})
                else:
                    dictionary.update({'port':sys.argv[2]})

    running_program(**dictionary)

def running_program(**kwargs):
    from app import app
    from waitress import serve

    serve(app, host=kwargs['host'], port=kwargs['port'])



