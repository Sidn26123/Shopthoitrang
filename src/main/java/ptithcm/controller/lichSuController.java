package ptithcm.controller;

import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import javax.transaction.Transactional;

import org.hibernate.SessionFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.RequestMapping;

import ptithcm.entity.DonHangEntity;
import ptithcm.entity.NguoiDungEntity;
import ptithcm.service.DonHangService;

@Transactional
@Controller
public class lichSuController {
	@Autowired
	SessionFactory factory;
	
	@Autowired 
	DonHangService donHangService;
	@RequestMapping("/lich-su-mua-hang")
	public String showHistory(HttpServletRequest request, ModelMap model) {
		HttpSession session0 = request.getSession();
		NguoiDungEntity user = (NguoiDungEntity) session0.getAttribute("USER");
		if (user == null) {
			model.addAttribute("user", new NguoiDungEntity());

			return "/user/login";
		}
		List<DonHangEntity> donHangChoXacNhanList = donHangService. timDonHangCuaUserTheoTrangThai(user.getMaNd(), 1);
		List<DonHangEntity> donHangDangGiaoList = donHangService. timDonHangCuaUserTheoTrangThai(user.getMaNd(), 2);
		List<DonHangEntity> donHangThanhCongList = donHangService. timDonHangCuaUserTheoTrangThai(user.getMaNd(), 3);
		List<DonHangEntity> donHangDaHuyList = donHangService. timDonHangCuaUserTheoTrangThai(user.getMaNd(), 0);
		model.addAttribute("user", user);
		model.addAttribute("donHangChoXacNhanList",donHangChoXacNhanList);
		model.addAttribute("donHangDangGiaoList",donHangDangGiaoList);
		model.addAttribute("donHangThanhCongList",donHangThanhCongList);	
		model.addAttribute("donHangDaHuyList",donHangDaHuyList);
		return "history/orderHistory";
	}
	
	
}
