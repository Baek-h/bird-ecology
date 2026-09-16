"""Capture only the application's own window, even when other apps obscure it."""
import ctypes
from ctypes import wintypes as W
from PIL import Image

def capture_widget(widget,path):
 widget.update_idletasks()
 user=ctypes.windll.user32; gdi=ctypes.windll.gdi32
 hwnd=widget.winfo_id()
 user.GetDC.argtypes=[W.HWND];user.GetDC.restype=W.HDC
 user.ReleaseDC.argtypes=[W.HWND,W.HDC]
 user.GetClientRect.argtypes=[W.HWND,ctypes.POINTER(W.RECT)]
 user.PrintWindow.argtypes=[W.HWND,W.HDC,W.UINT];user.PrintWindow.restype=W.BOOL
 gdi.CreateCompatibleDC.argtypes=[W.HDC];gdi.CreateCompatibleDC.restype=W.HDC
 gdi.CreateCompatibleBitmap.argtypes=[W.HDC,ctypes.c_int,ctypes.c_int];gdi.CreateCompatibleBitmap.restype=W.HBITMAP
 gdi.SelectObject.argtypes=[W.HDC,W.HANDLE];gdi.SelectObject.restype=W.HANDLE
 gdi.DeleteObject.argtypes=[W.HANDLE];gdi.DeleteDC.argtypes=[W.HDC]
 rect=W.RECT();user.GetClientRect(hwnd,ctypes.byref(rect));width=rect.right;height=rect.bottom
 class Header(ctypes.Structure):
  _fields_=[('size',W.DWORD),('width',W.LONG),('height',W.LONG),('planes',W.WORD),('bits',W.WORD),('compression',W.DWORD),('imageSize',W.DWORD),('xppm',W.LONG),('yppm',W.LONG),('used',W.DWORD),('important',W.DWORD)]
 class Info(ctypes.Structure):_fields_=[('header',Header),('colors',W.DWORD*3)]
 info=Info();info.header=Header(ctypes.sizeof(Header),width,-height,1,32,0,width*height*4,0,0,0,0)
 gdi.GetDIBits.argtypes=[W.HDC,W.HBITMAP,W.UINT,W.UINT,ctypes.c_void_p,ctypes.POINTER(Info),W.UINT]
 dc=user.GetDC(hwnd); memory=gdi.CreateCompatibleDC(dc);bitmap=gdi.CreateCompatibleBitmap(dc,width,height);old=gdi.SelectObject(memory,bitmap)
 try:
  if not user.PrintWindow(hwnd,memory,3):raise RuntimeError('Application window capture failed')
  buffer=ctypes.create_string_buffer(width*height*4)
  if not gdi.GetDIBits(memory,bitmap,0,height,buffer,ctypes.byref(info),0):raise RuntimeError('Window bitmap read failed')
  Image.frombuffer('RGB',(width,height),buffer.raw,'raw','BGRX',0,1).save(path)
 finally:
  gdi.SelectObject(memory,old);gdi.DeleteObject(bitmap);gdi.DeleteDC(memory);user.ReleaseDC(hwnd,dc)
