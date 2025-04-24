package ptithcm.controller;

import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import javax.transaction.Transactional;

import org.hibernate.SessionFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.ModelMap;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

import ptithcm.entity.NguoiDungEntity;
import ptithcm.entity.YeuThichEntity;
import ptithcm.service.yeuThichService;

@Transactional
@Controller
public class yeuThichController {
	@Autowired
	SessionFactory factory;
	@Autowired
	yeuThichService yeuThichService;
	
	@RequestMapping("yeuThich")
	public String showYeuThich(HttpServletRequest request, ModelMap model) {
		HttpSession session0 = request.getSession();
		
		NguoiDungEntity user = (NguoiDungEntity) session0.getAttribute("USER");
		if (user == null) {
			model.addAttribute("user", new NguoiDungEntity());

			return "/user/login";
		}
		List<YeuThichEntity> yeuThichList = yeuThichService.layDSYeuThichCuaUser(user.getMaNd());
		model.addAttribute("yeuThichList", yeuThichList);	
		return "yeuThich/yeuThich";
	}
	@RequestMapping(value="xoaYeuThich/{maYT}")
	public String deleteYeuThich(@PathVariable("maYT") int maYT, HttpServletRequest request, ModelMap model) {
		yeuThichService.deleteYeuThich(maYT);
		
		return "redirect:/yeuThich.htm";
	}

	
}
