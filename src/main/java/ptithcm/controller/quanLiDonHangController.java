package ptithcm.controller;

import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import javax.transaction.Transactional;

import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import ptithcm.entity.DonHangEntity;
import ptithcm.entity.NguoiDungEntity;
import ptithcm.service.DonHangService;

@Transactional
@Controller
public class quanLiDonHangController {
	
	@Autowired
	SessionFactory factory;
	
	@Autowired
	DonHangService donHangService;
	
	@RequestMapping(value="admin/order", method = RequestMethod.GET)
	public String product(ModelMap model, HttpServletRequest request) {
		HttpSession session0 = request.getSession();
		NguoiDungEntity user = (NguoiDungEntity) session0.getAttribute("USER");
		
		List<DonHangEntity> listDonHang = donHangService.layAllDonHang();
		model.addAttribute("listDonHang", listDonHang);
		
		return "admin/order";
	}
	
	@RequestMapping(value = "admin/order", params = "changeStatus2", method = RequestMethod.POST)
	public String chuyenDonHangSangDangGiao(HttpServletRequest request) {
	    int maDonHang = Integer.parseInt(request.getParameter("maDonHangDuyet"));
	    int trangThai = Integer.parseInt(request.getParameter("trangThaiDonDuyet"));

	    if (trangThai == 1) { // Nếu trạng thái là "Chờ xác nhận"
	        DonHangEntity donHang = donHangService.timDonHangTheoMa(maDonHang);
	        donHang.setTrangThai(2); // Chuyển trạng thái đơn hàng sang "Đang giao"
	        donHangService.updateDonHang(donHang);
	    }

	    return "redirect:/admin/order.htm";
	}
	
	@RequestMapping(value = "admin/order", params = "changeStatus0", method = RequestMethod.POST)
	public String chuyenDonHangSangHuy(HttpServletRequest request) {
	    int maDonHang = Integer.parseInt(request.getParameter("maDonHangHuy"));
	    int trangThai = Integer.parseInt(request.getParameter("trangThaiDonHuy"));

	    if (trangThai == 1) { // Nếu trạng thái là "Chờ xác nhận"
	        DonHangEntity donHang = donHangService.timDonHangTheoMa(maDonHang);
	        donHang.setTrangThai(0); // Chuyển trạng thái đơn hàng sang "Đang giao"
	        donHangService.updateDonHang(donHang);
	    }

	    return "redirect:/admin/order.htm";
	}

	
}
