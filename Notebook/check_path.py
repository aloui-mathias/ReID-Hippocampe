import imghdr
import sys


def main():
    file_list = []
    with open(sys.argv[1]) as f:
        for name in f.readlines():
            name = name.replace('\n', '')
            try:
                img_type = name.split('.')[-1]
                if imghdr.what(name) in ['jpeg', img_type]:
                    print(name)
                    file_list.append(name)
            except FileNotFoundError:
                # print(name)
                continue
    with open(sys.argv[1], "w") as f:
        for name in file_list:
            f.write(name + '\n')


if __name__ == "__main__":
    main()
