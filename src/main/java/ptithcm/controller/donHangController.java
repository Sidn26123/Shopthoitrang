package ptithcm.controller;

import java.util.Date;
import java.util.Iterator;
import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import ptithcm.entity.CTDonHangEntity;
import ptithcm.entity.DanhGiaEntity;
import ptithcm.entity.DonHangEntity;
import ptithcm.entity.GioHangEntity;
import ptithcm.entity.NguoiDungEntity;
import ptithcm.entity.SanPhamEntity;
import ptithcm.service.CTDonHangService;
import ptithcm.service.DanhGiaService;
import ptithcm.service.DonHangService;
import ptithcm.service.SanPhamService;
import ptithcm.service.gioHangService;

@Controller

public class donHangController {

	@Autowired
	DonHangService DonHangService;
	@Autowired
	gioHangService gioHangService;
	@Autowired
	CTDonHangService ctDonHangService;
	@Autowired
	SanPhamService sanPhamService;
	@Autowired
	DanhGiaService danhGiaService;
	@RequestMapping("donHang")
	public String donHang(HttpServletRequest request, ModelMap model) {
		HttpSession session = request.getSession();
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USER");

		List<GioHangEntity> gioHangList = gioHangService.layGioHangCuaUser(user.getMaNd());
		model.addAttribute("gioHangList", gioHangList);

		return "/donHang/donHang";

	}

	@RequestMapping("donHang/newInfo")
	public String donHang1(HttpServletRequest request, ModelMap model) {
		HttpSession session = request.getSession();
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USER");

		List<GioHangEntity> gioHangList = gioHangService.layGioHangCuaUser(user.getMaNd());
		model.addAttribute("gioHangList", gioHangList);
		System.out.print("test");
		return "/donHang/donHang";

	}
	@RequestMapping(value = "donHang/newInfo", method = RequestMethod.POST)
	public String newInfo(HttpServletRequest request) {
		HttpSession session = request.getSession();
		NguoiDungEntity newInfo = new NguoiDungEntity();
		String ten = request.getParameter("ten");
		String sdt = request.getParameter("sdt");
		String diaChi = request.getParameter("diaChi");
		newInfo.setHoTen(ten);
		newInfo.setSdt(sdt);
		newInfo.setDiaChi(diaChi);
		session.setAttribute("NEWINFO", newInfo);
System.out.print("newinfo");
		return "redirect:/donHang.htm";
	}

	@RequestMapping(value = "donHang/datHang", method = RequestMethod.POST, params = "datHang")
	public String datHang(HttpServletRequest request, HttpSession session) {

	    NguoiDungEntity newInfo = (NguoiDungEntity) session.getAttribute("NEWINFO"); // lấy địa chỉ mới
	    NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USER");
	    List<GioHangEntity> gioHangList = gioHangService.layGioHangCuaUser(user.getMaNd());
	    int tongTien = Integer.parseInt(request.getParameter("tongTien"));
	    DonHangEntity donHang = new DonHangEntity();
	    Date currentDate = new Date();
	    donHang.setNgayTao(currentDate);
	    donHang.setTongTien(tongTien);
	    donHang.setTrangThai(1);
	    if (newInfo == null) {
	        donHang.setHoTen(user.getHoTen());
	        donHang.setSdt(user.getSdt());
	        donHang.setDiaChi(user.getDiaChi());

	        donHang.setNguoiDung(user);

	    } else {
	        donHang.setHoTen(newInfo.getHoTen());
	        donHang.setSdt(newInfo.getSdt());
	        donHang.setDiaChi(newInfo.getDiaChi());
	        donHang.setNguoiDung(user);
	    }
	    DonHangService.luuDonHang(donHang);

	    GioHangEntity gh = new GioHangEntity();
	    SanPhamEntity sp = new SanPhamEntity();

	    for (int i = 0; i < gioHangList.size(); i++) {
	        gh = gioHangList.get(i);
	        CTDonHangEntity ctdh = new CTDonHangEntity();
	        ctdh.setDonHang(donHang);
	        ctdh.setSanPham(gh.getSanPham());
	        ctdh.setDonGia(gh.getSanPham().getDonGia());
	        ctdh.setSoLuong(gh.getSoLuong());
	        ctdh.setSize(gh.getSize()); // Set the size information
	        ctdh.setTrangThaiDanhGia(false);
	        ctDonHangService.luuCtdh(ctdh);

	        sp = sanPhamService.laySanPham(gh.getSanPham().getMaSP());
	        sp.setSoLuong(sp.getSoLuong() - gh.getSoLuong());
	        sanPhamService.updateSanPham(sp);
	        gioHangService.deleteGioHang(gh.getMaGh());

	    }

	    return "/donHang/thanhCong";

	}


	@RequestMapping("diaChiMacDinh")
	public String macDinh(HttpServletRequest request) {
		HttpSession session = request.getSession();
		session.removeAttribute("NEWINFO");
		return "redirect:/donHang.htm";
	}
	@RequestMapping("chiTietDonHang/{maDh}")
	public String ctDonHang(@PathVariable("maDh")int maDh, ModelMap model, HttpSession session )
	{
		NguoiDungEntity user = (NguoiDungEntity) session.getAttribute("USER");
		if (user == null) {
			model.addAttribute("user", new NguoiDungEntity());

			return "/user/login";
		}
		DonHangEntity donHang=DonHangService.timDonHangTheoMa(maDh);
		List<CTDonHangEntity> ctDonHangList=ctDonHangService.timctdhTheoMaDh(maDh);
		model.addAttribute("donHang",donHang);
		model.addAttribute("ctDonHangList",ctDonHangList);
		return "donHang/chiTietDonHang";
	}
	
	
	
	@RequestMapping("daNhanHang/{maDh}")
	public String daNhanHang(@PathVariable("maDh")int maDh )
	{
		DonHangEntity donHang=DonHangService.timDonHangTheoMa(maDh);
		donHang.setTrangThai(3);
		DonHangService.updateDonHang(donHang);
		
		return "redirect:/lich-su-mua-hang.htm";
	}
	
	@RequestMapping("huyDonHang/{maDh}")
	public String huyDonHang(@PathVariable("maDh")int maDh )
	{
		DonHangEntity donHang=DonHangService.timDonHangTheoMa(maDh);
		donHang.setTrangThai(0);
		DonHangService.updateDonHang(donHang);
		
		List<CTDonHangEntity> ctDonHangList=ctDonHangService.timctdhTheoMaDh(maDh);
		 SanPhamEntity sp=new SanPhamEntity();
		 for (int i=0; i<ctDonHangList.size();i++)
		 {
			 sp=sanPhamService.laySanPham(ctDonHangList.get(i).getSanPham().getMaSP());
			 sp.setSoLuong(sp.getSoLuong()+ctDonHangList.get(i).getSoLuong());
			 sanPhamService.updateSanPham(sp);
		 }
		
		return "redirect:/lich-su-mua-hang.htm";
	}
	

}