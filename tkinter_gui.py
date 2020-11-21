# モジュールのインポート
import os, tkinter, tkinter.filedialog, tkinter.messagebox
import numpy as np
import cv2
import main


def file_maker(fomat):
    # ファイル選択ダイアログの表示
    root = tkinter.Tk()
    root.withdraw()

    # ここの1行を変更　fTyp = [("","*")] →　fTyp = [("","*.csv")]
    fTyp = [("", "*." + fomat)]

    iDir = os.path.abspath(os.path.dirname(__file__))
    tkinter.messagebox.showinfo('open', '処理ファイルを選択してください！')
    file = tkinter.filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)

    # 処理ファイル名の出力(
    # tkinter.messagebox.showinfo('出力ファイル',file)
    return file
