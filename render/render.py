#!/usr/bin/env python3

import numpy as np
from numpy.random import random

import cairo

import gi
gi.require_version('Gtk','3.0')
from gi.repository import Gtk

tau = np.pi * 2

class Render(object):

  def __init__(self,n=512,fg=[0,0,0,1],bg=[1,1,1,1]):
    self.n = n
    self.fg = fg
    self.bg = bg
    self.pix = 1./float(n)

    self.num_img = 0

    self.__init_cairo()

  def __init_cairo(self):
    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.n,self.n)
    ctx = cairo.Context(sur)
    ctx.scale(self.n,self.n)

    self.sur = sur
    self.ctx = ctx

    self.clear_canvas()

  def clear_canvas(self):
    ctx = self.ctx

    ctx.set_source_rgba(*self.bg)
    ctx.rectangle(0,0,1,1)
    ctx.fill()
    ctx.set_source_rgba(*self.fg)

  def write_to_png(self,fn):
    self.sur.write_to_png(fn)
    self.num_img += 1

  def set_fg(self,c):
    self.fg = c
    self.ctx.set_source_rgba(*c)

  def set_bg(self,c):
    self.bg = c

  def set_line_width(self,w):
    self.line_width = w
    self.ctx.set_line_width(w)

  def line(self,x1,y1,x2,y2):
    ctx = self.ctx
    ctx.move_to(x1,y1)
    ctx.line_to(x2,y2)
    ctx.stroke()

  def triangle(self,x1,y1,x2,y2,x3,y3,fill=False):
    ctx = self.ctx
    ctx.move_to(x1,y1)
    ctx.line_to(x2,y2)
    ctx.line_to(x3,y3)
    ctx.close_path()

    if fill:
      ctx.fill()
    else:
      ctx.stroke()

  def random_parallelogram(self,x1,y1,x2,y2,x3,y3,grains):
    pix = self.pix
    rectangle = self.ctx.rectangle
    fill = self.ctx.fill

    v1 = np.array((x2-x1,y2-y1))
    v2 = np.array((x3-x1,y3-y1))

    a1 = random((grains,1))
    a2 = random((grains,1))

    dd = v1*a1 + v2*a2

    dd[:,0] += x1
    dd[:,1] += y1

    for x,y in dd:
      rectangle(x,y,pix,pix)
      fill()

  def random_triangle(self,x1,y1,x2,y2,x3,y3,grains):
    pix = self.pix
    rectangle = self.ctx.rectangle
    fill = self.ctx.fill

    v1 = np.array((x2-x1,y2-y1))
    v2 = np.array((x3-x1,y3-y1))

    a1 = random((2*grains,1))
    a2 = random((2*grains,1))

    mask = ((a1+a2)<1).flatten()

    # discarding half the grains because i am too tired to figure out how to
    # map the parallelogram to the triangle

    dd = v1*a1 + v2*a2

    dd[:,0] += x1
    dd[:,1] += y1

    for x,y in dd[mask,:]:
      rectangle(x,y,pix,pix)
      fill()

  def dot(self,x,y):
    ctx = self.ctx
    pix = self.pix
    ctx.rectangle(x,y,pix,pix)
    ctx.fill()

  def circle(self,x,y,r,fill=False):
    ctx = self.ctx

    ctx.arc(x,y,r,0,tau)
    if fill:
      ctx.fill()
    else:
      ctx.stroke()

  def random_circle(self,x1,y1,r,grains):
    """
    random points in circle. nonuniform distribution.
    """
    pix = self.pix
    rectangle = self.ctx.rectangle
    fill = self.ctx.fill

    the = random(grains)*np.pi*2
    rad = random(grains)*r

    xx = x1 + np.cos(the)*rad
    yy = y1 + np.sin(the)*rad

    for x,y in zip(xx,yy):
      rectangle(x,y,pix,pix)
      fill()

  def random_uniform_circle(self,x1,y1,r,grains,dst=0):
    from helpers import darts

    pix = self.pix
    rectangle = self.ctx.rectangle
    fill = self.ctx.fill

    for x,y in darts(grains,x1,y1,r,dst):
      rectangle(x,y,pix,pix)
      fill()

  def transparent_pix(self):
    op = self.ctx.get_operator()
    self.ctx.set_operator(cairo.OPERATOR_SOURCE)
    self.ctx.set_source_rgba(*[1,1,1,0.95])
    self.dot(1-self.pix,1.0-self.pix)
    self.ctx.set_operator(op)

  def path(self,xy):
    ctx = self.ctx
    ctx.move_to(*xy[0,:])
    for x in xy:
      ctx.line_to(*x)
    ctx.stroke()

  def closed_path(self,coords,fill=True):
    ctx = self.ctx
    line_to = ctx.line_to

    x,y = coords[0]
    ctx.move_to(x,y)

    for x,y in coords[1:]:
      line_to(x,y)

    ctx.close_path()

    if fill:
      ctx.fill()
    else:
      ctx.stroke()

  def circle_path(self,coords,r,fill=False):
    ctx = self.ctx
    for x,y in coords:
      ctx.arc(x,y,r,0,tau)
      if fill:
        ctx.fill()
      else:
        ctx.stroke()

  def circles(self,x1,y1,x2,y2,r,nmin=2):
    arc = self.ctx.arc
    fill = self.ctx.fill

    dx = x1-x2
    dy = y1-y2
    dd = np.sqrt(dx*dx+dy*dy)

    n = int(dd/self.pix)
    n = n if n>nmin else nmin

    a = np.arctan2(dy,dx)

    scale = np.linspace(0,dd,n)

    xp = x1-scale*np.cos(a)
    yp = y1-scale*np.sin(a)

    for x,y in zip(xp,yp):
      arc(x,y,r,0,np.pi*2.)
      fill()

  def sandstroke_orthogonal(self,xys,height=None,steps=10,grains=10):
    pix = self.pix
    rectangle = self.ctx.rectangle
    fill = self.ctx.fill

    if not height:
      height = pix*10

    dx = xys[:,2] - xys[:,0]
    dy = xys[:,3] - xys[:,1]

    aa = np.arctan2(dy,dx)
    directions = np.column_stack([np.cos(aa),np.sin(aa)])
    dd = np.sqrt(np.square(dx)+np.square(dy))

    aa_orth = aa + np.pi*0.5
    directions_orth = np.column_stack([np.cos(aa_orth),np.sin(aa_orth)])

    for i,d in enumerate(dd):

      xy_start = xys[i,:2] + \
          directions[i,:]*random((steps,1))*d

      for xy in xy_start:
        points = xy + \
            directions_orth[i,:]*random((grains,1))*height
        for x,y in points:
          rectangle(x,y,pix,pix)
          fill()

  def sandstroke_non_linear(self,xys,grains=10,left=True):
    pix = self.pix
    rectangle = self.ctx.rectangle
    fill = self.ctx.fill

    dx = xys[:,2] - xys[:,0]
    dy = xys[:,3] - xys[:,1]

    aa = np.arctan2(dy,dx)
    directions = np.column_stack([np.cos(aa),np.sin(aa)])

    dd = np.sqrt(np.square(dx)+np.square(dy))

    for i,d in enumerate(dd):
      rnd = np.sqrt(random((grains,1)))
      if left:
        rnd = 1.0-rnd

      for x,y in xys[i,:2] + directions[i,:]*rnd*d:
        rectangle(x,y,pix,pix)
        fill()

  def sandstroke(self,xys,grains=10):
    pix = self.pix
    rectangle = self.ctx.rectangle
    fill = self.ctx.fill

    dx = xys[:,2] - xys[:,0]
    dy = xys[:,3] - xys[:,1]

    aa = np.arctan2(dy,dx)
    directions = np.column_stack([np.cos(aa),np.sin(aa)])

    dd = np.sqrt(np.square(dx)+np.square(dy))

    for i,d in enumerate(dd):
      for x,y in xys[i,:2] + directions[i,:]*random((grains,1))*d:
        rectangle(x,y,pix,pix)
        fill()

class Animate(Render):

  def __init__(self,callback,*args,vsync=False,**kwargs):
    super().__init__(*args,**kwargs)

    self.steps = 0

    window = Gtk.Window()
    self.window = window

    window.set_title("Animate")
    window.set_default_size(self.n,self.n)
    window.connect('delete-event',Gtk.main_quit)

    self.callback = callback

    darea = Gtk.DrawingArea()
    self.darea = darea

    darea.connect("draw",self._draw)

    window.add(darea)
    window.show_all()

    if vsync:
        window.add_tick_callback(self.update)
    else:
        from gi.repository import GLib
        GLib.idle_add(self.update)

    Gtk.main() # NOTE: formerly in self.start()

  def __destroy(self,*args):
    print("(DEBUG)","destroy called with",args)
    Gtk.main_quit(*args)

  def _draw(self,widget,cr):
    cr.set_source_surface(self.sur,0,0)
    cr.paint()

  def update(self,*args):
    res = self.callback(self)
    self.steps += 1
    self.darea.queue_draw()
    return res

if __name__ == '__main__':
    stop = 5e3
    def draw(anim):
        t = anim.steps/stop

        if t >= 1:
            return False # stop drawing

        if t == 0:
            anim.set_line_width(anim.pix)
            anim.line(0.0,0.0,1.0,1.0)
            anim.line(0.5,0.5,1.0,0.5)

        anim.random_circle(t,1-t,0.25,16)

        return True # keep drawing

    Animate(draw)
