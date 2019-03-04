import configparser

config = configparser.ConfigParser()

config.read("modules/altair_config.ini")
#config.read("junk.ini")


print(config.sections())

print(config["data_dirs"]["DIR_FYI_INFO"])
