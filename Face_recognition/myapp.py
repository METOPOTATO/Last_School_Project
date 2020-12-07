from tkinter import Tk,Frame, Button,Label
from myapp_pages import MainPage,PageManage
import cv2
class App(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.geometry('700x700')
        container = Frame(self)
        container.pack()

        self.frames = {}
        for F in (MainPage, PageManage):
            page_name = F.__name__  
            self.frames[page_name] = F(parent=container,controller=self)
            self.frames[page_name].grid(row=0,column=0,sticky='nsew')

        self.show_frame('MainPage')
    
    def show_frame(self,page_name):
        try:
            if page_name == 'MainPage':
                self.frames[page_name].run_video()
                self.frames[page_name].video_loop()
                print(self.frames[page_name].winfo_height())
                print(self.frames[page_name].winfo_width())
                # self.frames['PageManage'].hide()

            if page_name == 'PageManage':
                self.frames['PageManage'].show()
        except:
            pass
        frame = self.frames[page_name]
        frame.tkraise()



if __name__ == "__main__":
    app = App()
    app.mainloop()