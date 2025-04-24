package ptithcm.controller;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.sql.Driver;
import java.util.Date;
import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

import ptithcm.entity.CTDonHangEntity;
import ptithcm.entity.DanhGiaEntity;
import ptithcm.entity.DonHangEntity;
import ptithcm.entity.NguoiDungEntity;
import ptithcm.entity.SanPhamEntity;
import ptithcm.service.CTDonHangService;
import ptithcm.service.DanhGiaService;
import ptithcm.service.DonHangService;
import ptithcm.service.SanPhamService;

@Controller

public class danhGiaController {
	@Autowired
	DonHangService donHangService;
	@Autowired
	CTDonHangService ctDonHangService;
	@Autowired
	SanPhamService sanPhamService;
	@Autowired
	DanhGiaService danhGiaService;

	@RequestMapping("danhGia/{maDh}")
	public String danhGia(@PathVariable("maDh") int maDh, ModelMap model, HttpSession session) {
		session.setAttribute("MADH", maDh);
		DonHangEntity donHang = donHangService.timDonHangTheoMa(maDh);
		List<CTDonHangEntity> ctDonHangList = ctDonHangService.timctdhTheoMaDh(maDh);
		model.addAttribute("donHang", donHang);
		model.addAttribute("ctDonHangList", ctDonHangList);

		return "donHang/danhGia";
	}
	@RequestMapping(value="chiTietDonHang/{maDh}/{maSp}", params="submit-rating")
	public String guiDanhGia(@PathVariable("maDh") String maDh, @PathVariable("maSp") String maSp, HttpServletRequest request, ModelMap model ) throws IOException {
		HttpSession session0 = request.getSession();
		NguoiDungEntity user = (NguoiDungEntity) session0.getAttribute("USER");
		if (user == null) {
			model.addAttribute("user", new NguoiDungEntity());

			return "/user/login";
		}
		int maDHTypeInt = Integer.parseInt(maDh);
		DonHangEntity donHang = donHangService.timDonHangTheoMa(maDHTypeInt);
		List<CTDonHangEntity> ctDonHangList=ctDonHangService.timctdhTheoMaDh(maDHTypeInt);
		Date currentDate = new Date();
		DanhGiaEntity danhGia = new DanhGiaEntity();
		String content = request.getParameter("content-danhGia");
		float soSao = Float.parseFloat(request.getParameter("rate"));
		SanPhamEntity sanpham = sanPhamService.laySanPham(maSp);
		CTDonHangEntity ctdh= ctDonHangService.timCtdhTheoMaDhMaSP(maDHTypeInt, maSp);
	
		danhGia.setNgay(currentDate);
		danhGia.setNguoiDung(user);
		danhGia.setNoiDung(content);
		danhGia.setSoSao(soSao);
		danhGia.setSanPham(sanpham);
		danhGiaService.saveDanhGia(danhGia);
		System.out.println("luu danh gia");
		ctdh.setTrangThaiDanhGia(true);
		ctDonHangService.updateCtdh(ctdh);
		
		float soSaoTB = sanPhamService.tinhSoSaoTB(sanpham);
	    sanpham.setSoSaoTB(soSaoTB);
	    sanPhamService.updateSanPham(sanpham);
	    
	    ProcessBuilder builder = new ProcessBuilder(
                "cmd.exe", "/c", "cd C:\\Users\\Administrator\\Documents\\ShopThoiTrang\\src\\main\\python & python recommendTrain.py");
        builder.redirectErrorStream(true);
        Process p = builder.start();
        BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream()));
        System.out.println("test train");
        String line;
        while ((line = r.readLine()) != null) {
        	System.out.println(line);
        }
		
		return "redirect:/chiTietDonHang/"+maDh+".htm";

	}

	/*
	 * @RequestMapping("luuDanhGia/{maSp}/{maCTDH}") public String
	 * luuDanhGia(@PathVariable("maSp") String maSp, @PathVariable("maCTDH") int
	 * maCTDH, HttpServletRequest request, HttpSession session) { Date currentDate =
	 * new Date(); DanhGiaEntity danhGia = new DanhGiaEntity(); NguoiDungEntity user
	 * = (NguoiDungEntity) session.getAttribute("USER"); SanPhamEntity sanPham =
	 * sanPhamService.laySanPham(maSp); CTDonHangEntity ctdh =
	 * ctDonHangService.timCtdhTheoMaCtdh(maCTDH);
	 * 
	 * Float soSao = Float.parseFloat(request.getParameter("soSao")); String noiDung
	 * = request.getParameter("noiDung"); danhGia.setNguoiDung(user);
	 * danhGia.setNgay(currentDate); danhGia.setSoSao(soSao);
	 * danhGia.setNoiDung(noiDung); danhGia.setSanPham(sanPham);
	 * danhGiaService.saveDanhGia(danhGia);
	 * 
	 * ctdh.setTrangThaiDanhGia(true); ctDonHangService.updateCtdh(ctdh);
	 * 
	 * int maDh = (int) session.getAttribute("MADH");
	 * session.removeAttribute("MADH");
	 * 
	 * return "redirect:/danhGia/" + maDh + ".htm"; }
	 */

	@RequestMapping("luuDanhGia/{maSP}/{maDh}")
	public String luuDanhGia1(@PathVariable("maSP") String maSP, @PathVariable("maDh") String maDH, HttpServletRequest request, HttpSession session, ModelMap model) {
		int maDHTypeInt = Integer.parseInt(maDH);
		DonHangEntity donHang = donHangService.timDonHangTheoMa(maDHTypeInt);
		
		List<CTDonHangEntity> ctDonHangList=ctDonHangService.timctdhTheoMaDh(maDHTypeInt);
		model.addAttribute("donHang",donHang);
		model.addAttribute("ctDonHangList",ctDonHangList);
		
		return "donHang/chiTietDonHang";

	}

}